/**
 *Submitted for verification at Etherscan.io on 2017-11-23
*/

contract check {
    function add(address _add, uint _req) {
        _add.callcode(bytes4(keccak256("changeRequirement(uint256)")), _req);
    }
}